import numpy as np
import re

import sys

from openai import OpenAI
# api_key = ""
# api_base = ""
# client = OpenAI(api_key=api_key, base_url=api_base)

def gpt_api_no_stream(
    client,
    prompt: str, 
    model="gpt-4o",
    reset_messages: bool = True,
    response_only: bool = True
):
    """
    ------------
    Examples
    ------------
    
    try:
        response = gpt_api_no_stream(client, prompt, model=model)[1]
    except AuthenticationError:
        continue
    if "OpenAI API error" in response:
        print(f"{response}")
    else:
        np.save(savepath, response)
    """
    
    if "gpt-3.5" in model:
        model = "gpt-3.5-turbo-1106"
    elif "gpt-4omini" in model:
        model = "gpt-4o-mini-2024-07-18"
    elif "gpt-4o" in model:
        model = "gpt-4o-2024-11-20"
    elif "gpt-o1mini" in model:
        model = "o1-mini-2024-09-12"
    messages = [{'role': 'user','content': prompt},]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        completion = dict(completion)
        msg = None
        choices = completion.get('choices', None)
        if choices:
            msg = choices[0].message.content
        else:
            msg = completion.message.content
    except Exception as err:
        return (False, f'OpenAI API error: {err}')
    if reset_messages:
        messages.pop(-1)
    else:
        # add text of response to messages
        messages.append({
            'role': choices[0].message.role,
            'content': choices[0].message.content
        })
    if response_only:
        return True, msg
    else:
        return True, messages

def get_json_result(response):
    try:
        tem = response[::-1][response[::-1].index("}"):][::-1]
    
        cumulative = ""
        extra = 1
        while extra>0:
            extra -= 1
            idxb = tem[::-1].index("{")+1
            add = tem[::-1][:idxb][::-1]
            extra += np.array([a=="}" for a in list(add[1:-1])]).sum()
            cumulative = add + cumulative
            tem = tem[::-1][idxb:][::-1]
        curlyblankets = cumulative

        # ## Preprocessing
        pattern = r"//.*?\n" # delete comment-outs
        curlyblankets = re.sub(pattern, "", curlyblankets)
        l = [] # Delete "Target Text" and "Backchannel or not"
        alsonext = False
        for element in curlyblankets.split("\n"):
            if alsonext:
                alsonext = False
                continue
            if "Target Text" in element or "Backchannel" in element:
                if ":" in element[-2:]:
                    alsonext = True
                continue
            l += [element]
        curlyblankets = "\n".join(l)

        curlyblankets = curlyblankets.replace("null", '"neutral"')
        a = eval(curlyblankets)
    
    except (ValueError, SyntaxError, NameError):
        return False, None
    
    return True, a 